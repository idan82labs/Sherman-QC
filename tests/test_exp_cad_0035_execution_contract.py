from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import exp_cad_0035_execution_contract as contract


PREREG = Path(
    "docs/research/"
    "2026-08-30_exp_cad_0035_sequential_member_intake_preregistration.json"
)


def test_frozen_preregistration_identity_and_digest() -> None:
    block = PREREG.read_bytes()
    assert hashlib.sha256(block).hexdigest() == contract.PREREGISTRATION_SHA256
    value = json.loads(block)
    assert value["schema"] == "EXP_CAD_0035_SEQUENTIAL_MEMBER_INTAKE_PREREGISTRATION_V1"
    assert value["experiment_id"] == "EXP_CAD_0035"
    assert value["preregistration_boundary"]["physical_scan_hash_call_count_during_exp35_design"] == 0
    assert value["preregistration_boundary"]["physical_scan_point_cloud_load_call_count_during_exp35_design"] == 0
    assert value["preregistration_boundary"]["physical_scan_byte_open_operation_count_during_exp35_design"] == 0


def test_scan_rows_are_expected_constants_not_file_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))

    def forbidden_hash(_path: Path) -> str:
        raise AssertionError("physical scan hash must not run while constructing lock rows")

    def forbidden_metadata(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("physical scan metadata must remain untouched in lock rows")

    monkeypatch.setattr(contract, "sha256_path", forbidden_hash)
    monkeypatch.setattr(contract.Path, "exists", forbidden_metadata)
    monkeypatch.setattr(contract.Path, "stat", forbidden_metadata)
    monkeypatch.setattr(contract.Path, "resolve", forbidden_metadata)
    monkeypatch.setattr(contract.Path, "is_symlink", forbidden_metadata)
    rows = contract._physical_scan_rows(prereg)
    assert set(rows) == {"scan_N009", "scan_N010"}
    assert all(row["bytes_opened_or_hashed_during_lock_construction"] is False for row in rows.values())
    assert rows["scan_N009"]["expected_sha256"] == "3567fd96870a93f39fcc1e203cef847bb44b72f5a5868e355693c1bfadd87833"
    assert rows["scan_N010"]["expected_sha256"] == "823e6a7dc7c42091c6565393c82122eea9fbed2bda3e079247189d463a28f1b3"
    assert "scan_N009" not in contract.REQUIRED_BOUND_FILES
    assert "scan_N010" not in contract.REQUIRED_BOUND_FILES


def test_null_canonical_publication_matches_parent_policy() -> None:
    value = {
        "nested": [float("nan"), np.float64("inf"), np.array([1.0, -np.inf])],
        "finite": -0.0,
    }
    assert contract.canonical_bytes(value) == (
        b'{"finite":-0.0,"nested":[null,null,[1.0,null]]}'
    )
    assert b"NaN" not in contract.canonical_bytes(value)
    assert b"Infinity" not in contract.canonical_bytes(value)


def test_atomic_writer_streams_without_materializing_canonical_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = {"z": [float("nan"), np.float64("inf"), *range(10_000)], "a": "x"}
    expected = contract.canonical_bytes(value)

    def forbidden(_value: object) -> bytes:
        raise AssertionError("streaming writer must not materialize canonical_bytes")

    monkeypatch.setattr(contract, "canonical_bytes", forbidden)
    path = tmp_path / "streamed.json"
    digest = contract.canonical_json_write(path, value)
    assert path.read_bytes() == expected
    assert hashlib.sha256(expected).hexdigest() == digest
    assert value["z"][:2] == [None, None]


def test_authorized_roots_are_exact_and_pairwise_disjoint() -> None:
    rows = contract._authorized_output_rows()
    roots = [
        Path(rows[key])
        for key in (
            "primary_source",
            "replay_source",
            "primary_measurement",
            "replay_measurement",
            "primary_renderer",
            "replay_renderer",
        )
    ]
    for index, first in enumerate(roots):
        for second in roots[index + 1 :]:
            assert first != second
            assert first not in second.parents
            assert second not in first.parents
    assert rows == contract.AUTHORIZED_OUTPUTS
    assert rows["primary_lifecycle_measurement"].endswith(
        "/primary/measurement/SOURCE_LIFECYCLE_MEASUREMENT.json"
    )
    assert rows["replay_lifecycle_measurement"].endswith(
        "/replay/measurement/SOURCE_LIFECYCLE_MEASUREMENT.json"
    )


def test_qualification_artifact_inventory_is_exact_and_canonical() -> None:
    assert len(contract.QUALIFICATION_ARTIFACT_PATHS) == 18
    assert set(contract.QUALIFICATION_ARTIFACT_PATHS) <= contract.REQUIRED_BOUND_FILES
    assert contract.QUALIFICATION_ARTIFACT_PATHS["fixture_manifest"].endswith(
        "/EXP_CAD_0035_SOURCE_FREE_FIXTURE_MANIFEST.json"
    )
    assert contract.QUALIFICATION_ARTIFACT_PATHS["qualification_aggregate"].endswith(
        "/EXP_CAD_0035_SOURCE_FREE_MEMORY_QUALIFICATION_AGGREGATE.json"
    )
    trial_results = [
        path
        for key, path in contract.QUALIFICATION_ARTIFACT_PATHS.items()
        if key.startswith("qualification_trial_") and key.endswith("_result")
    ]
    trial_measurements = [
        path
        for key, path in contract.QUALIFICATION_ARTIFACT_PATHS.items()
        if key.startswith("qualification_trial_") and key.endswith("_measurement")
    ]
    assert len(trial_results) == len(trial_measurements) == 6


def test_lock_command_templates_match_fresh_orchestrator_and_source_parser() -> None:
    orchestrator_path = (
        contract.ROOT
        / "scripts/orchestrate_exp_cad_0035_cfpdg_v5_sequential_physical.py"
    )
    source_path = contract.ROOT / "scripts/run_exp_cad_0035_cfpdg_source.py"
    orch_spec = importlib.util.spec_from_file_location("exp35_template_orch", orchestrator_path)
    source_spec = importlib.util.spec_from_file_location("exp35_template_source", source_path)
    assert orch_spec and orch_spec.loader and source_spec and source_spec.loader
    orchestrator = importlib.util.module_from_spec(orch_spec)
    source = importlib.util.module_from_spec(source_spec)
    orch_spec.loader.exec_module(orchestrator)
    source_spec.loader.exec_module(source)
    assert tuple(name.replace("_", "-") for name in orchestrator.PATH_ARGUMENTS) == (
        contract.ORCHESTRATOR_PATH_ARGUMENTS
    )
    parser_flags = tuple(
        action.option_strings[0].removeprefix("--")
        for action in source.build_parser()._actions
        if action.option_strings and action.option_strings[0] != "-h"
    )
    assert parser_flags == contract.SOURCE_ARGUMENT_ORDER
    renderer_command = orchestrator._renderer_command(
        SimpleNamespace(renderer=Path("/renderer.py")),
        source_root=Path("/source"),
        renderer_root=Path("/rendered"),
    )
    assert renderer_command[1:3] == ["-c", contract.RENDERER_BOOTSTRAP]
    environment = json.loads(
        Path(contract.ROOT / "docs/research/2026-08-30_exp_cad_0034_physical_environment_attestation.json").read_text(
            encoding="utf-8"
        )
    )["environment"]
    templates = contract.command_templates(environment)
    assert templates["source"]["required_flags_in_order"] == list(parser_flags)
    assert templates["renderer"]["argv_prefix"] == ["-c", renderer_command[2]]


def test_exact_cli_path_rejects_relative_parent_and_symlink_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "data" / "x.json"
    target.parent.mkdir()
    target.write_text("{}", encoding="utf-8")
    alias = root / "alias.json"
    alias.symlink_to(target)
    linked_parent = root / "linked"
    linked_parent.symlink_to(target.parent, target_is_directory=True)
    monkeypatch.setattr(contract, "ROOT", root)
    assert contract._exact_cli_path(target, "data/x.json", must_exist=True) == target
    with pytest.raises(ValueError):
        contract._exact_cli_path(Path("data/x.json"), "data/x.json", must_exist=True)
    with pytest.raises(ValueError):
        contract._exact_cli_path(root / "data" / ".." / "data" / "x.json", "data/x.json", must_exist=True)
    with pytest.raises(ValueError):
        contract._exact_cli_path(alias, "data/x.json", must_exist=True)
    with pytest.raises(ValueError, match="symlink"):
        contract._exact_cli_path(
            root / "linked" / "x.json", "linked/x.json", must_exist=True
        )
    with pytest.raises(ValueError, match="symlink"):
        contract._normalized_repo_relative("linked/x.json", must_exist=True)


def test_source_mode_requires_exact_primary_or_replay_paths() -> None:
    lock = {"authorized_outputs": contract.AUTHORIZED_OUTPUTS}
    for mode, prefix in (("PRIMARY", "primary"), ("REPLAY", "replay")):
        outputs = {
            generic: contract.ROOT / contract.AUTHORIZED_OUTPUTS[locked]
            for generic, locked in contract.SOURCE_MODE_OUTPUT_KEYS[mode].items()
        }
        assert contract.infer_source_mode(outputs, lock) == mode
    with pytest.raises(ValueError):
        contract.infer_source_mode({}, lock)


def test_test_attestation_fails_closed_on_skip_or_forbidden_access() -> None:
    test_ids = ["test_one"]
    original_count = contract.EXPECTED_SOURCE_FREE_TEST_COUNT
    original_digest = contract.EXPECTED_SOURCE_FREE_TEST_IDS_SHA256
    contract.EXPECTED_SOURCE_FREE_TEST_COUNT = 1
    contract.EXPECTED_SOURCE_FREE_TEST_IDS_SHA256 = hashlib.sha256(
        contract.canonical_bytes(test_ids)
    ).hexdigest()
    row = {
        "schema": contract.TEST_ATTESTATION_SCHEMA,
        "experiment_id": contract.EXPERIMENT_ID,
        "status": "PASS_SOURCE_FREE_TESTS",
        "physical_source_execution_authorized": False,
        "evaluator_input_authorized": False,
        "generalization_evidence": False,
        "result": {
            "collected_count": 1,
            "passed_count": 1,
            "failed_count": 0,
            "error_count": 0,
            "skipped_count": 0,
            "returncode": 0,
        },
        "forbidden_access_counts": {
            key: 0 for key in contract.FORBIDDEN_ACCESS_COUNT_KEYS
        },
        "collected_test_ids": test_ids,
        "tested_file_sha256": {"test": "0" * 64},
        "collect_command": list(contract.SOURCE_FREE_COLLECT_COMMAND),
        "test_command": list(contract.SOURCE_FREE_TEST_COMMAND),
    }
    try:
        contract._validate_test_attestation(row)
        skipped = json.loads(json.dumps(row))
        skipped["result"]["skipped_count"] = 1
        with pytest.raises(ValueError):
            contract._validate_test_attestation(skipped)
        opened = json.loads(json.dumps(row))
        opened["forbidden_access_counts"]["physical_scan_hash_call_count"] = 1
        with pytest.raises(ValueError):
            contract._validate_test_attestation(opened)
        empty_counters = json.loads(json.dumps(row))
        empty_counters["forbidden_access_counts"] = {}
        with pytest.raises(ValueError, match="inventory"):
            contract._validate_test_attestation(empty_counters)
        extra_counter = json.loads(json.dumps(row))
        extra_counter["forbidden_access_counts"]["unexpected"] = 0
        with pytest.raises(ValueError, match="inventory"):
            contract._validate_test_attestation(extra_counter)
        changed_command = json.loads(json.dumps(row))
        changed_command["test_command"].append("-x")
        with pytest.raises(ValueError, match="command"):
            contract._validate_test_attestation(changed_command)
    finally:
        contract.EXPECTED_SOURCE_FREE_TEST_COUNT = original_count
        contract.EXPECTED_SOURCE_FREE_TEST_IDS_SHA256 = original_digest


def test_environment_must_equal_frozen_exp34_environment() -> None:
    parent = json.loads(
        Path("docs/research/2026-08-30_exp_cad_0034_physical_environment_attestation.json").read_text(
            encoding="utf-8"
        )
    )
    row = {
        "schema": contract.ENVIRONMENT_ATTESTATION_SCHEMA,
        "experiment_id": contract.EXPERIMENT_ID,
        "status": "FROZEN",
        "physical_source_execution_authorized": False,
        "evaluator_input_authorized": False,
        "physical_scan_or_evaluator_input_used": False,
        "generalization_evidence": False,
        "environment": contract.runtime_environment_fingerprint(),
    }
    contract._validate_environment_attestation(row, parent)
    changed = json.loads(json.dumps(row))
    changed["environment"]["open3d_version"] = "different"
    with pytest.raises(ValueError):
        contract._validate_environment_attestation(changed, parent)


def test_live_runtime_fingerprint_matches_frozen_exp34_environment() -> None:
    parent = json.loads(
        Path("docs/research/2026-08-30_exp_cad_0034_physical_environment_attestation.json").read_text(
            encoding="utf-8"
        )
    )
    fingerprint = contract.runtime_environment_fingerprint()
    assert set(fingerprint) == set(contract.ENVIRONMENT_FINGERPRINT_KEYS)
    assert all(fingerprint[key] == value for key, value in parent["environment"].items())
    assert fingerprint["python_cache_tag"]
    assert fingerprint["python_soabi"]
    assert fingerprint["python_extension_suffix"].startswith(".")
    assert fingerprint["python_byteorder"] in {"little", "big"}
    assert fingerprint["python_pointer_bits"] in {"32", "64"}
    assert fingerprint["platform_system"]
    assert fingerprint["platform_machine"]


def test_test_attestation_binds_environment_attestation() -> None:
    assert "environment_attestation" in contract.TEST_ATTESTED_FILE_KEYS


def test_lock_builder_requires_exact_key_set_without_touching_inputs() -> None:
    with pytest.raises(ValueError, match="lock keys differ"):
        contract.build_lock_payload(bindings={}, lock_path=contract.ROOT / contract.LOCK_PATH)


def test_lock_builder_publication_is_durable_o_excl_and_handles_partial_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = contract.ROOT / "scripts/build_exp_cad_0035_physical_lock.py"
    spec = importlib.util.spec_from_file_location("exp35_lock_builder_test", path)
    builder = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(builder)
    output = tmp_path / "immutable.json"
    payload = {"schema": "TEST", "value": [1, 2, 3]}
    real_write = builder.os.write
    writes: list[int] = []

    def partial_write(descriptor: int, data: bytes) -> int:
        block = data[: max(1, len(data) // 2)]
        writes.append(len(block))
        return real_write(descriptor, block)

    monkeypatch.setattr(builder.os, "write", partial_write)
    digest = builder.create_immutable_lock(output, payload)
    assert output.read_bytes() == contract.canonical_bytes(payload)
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    assert len(writes) > 1
    with pytest.raises(FileExistsError):
        builder.create_immutable_lock(output, payload)


def _valid_detached_approval_fixture() -> tuple[dict[str, object], dict[str, object]]:
    digest = "a" * 64
    preregistration = (contract.ROOT / contract.PREREGISTRATION_PATH).resolve()
    bound = {
        key: {
            "path": contract.PREREGISTRATION_PATH,
            "size_bytes": preregistration.stat().st_size,
            "sha256": digest,
        }
        for key in contract.REQUIRED_BOUND_FILES
    }
    lock: dict[str, object] = {"bound_files": bound}
    row: dict[str, object] = {
        "schema": contract.DETACHED_APPROVAL_SCHEMA,
        "experiment_id": contract.EXPERIMENT_ID,
        "architecture": contract.ARCHITECTURE,
        "status": "APPROVED_FOR_ONE_EXP35_ORCHESTRATOR_INVOCATION",
        "preregistration_path": str(preregistration),
        "preregistration_sha256": contract.PREREGISTRATION_SHA256,
        "lock_path": "/immutable-lock.json",
        "lock_sha256": "b" * 64,
        "prelock_approval_path": str(preregistration),
        "prelock_approval_sha256": digest,
        "source_free_audit_packet_path": str(preregistration),
        "source_free_audit_packet_sha256": digest,
        "test_attestation_path": str(preregistration),
        "test_attestation_sha256": digest,
        "environment_attestation_path": str(preregistration),
        "environment_attestation_sha256": digest,
        "approved_bound_file_sha256": {
            key: digest for key in contract.REQUIRED_BOUND_FILES
        },
        "physical_source_execution_authorized": True,
        "evaluator_input_authorized": False,
        "authorized_orchestrator_invocations": 1,
        "maximum_primary_source_children": 1,
        "maximum_conditional_replay_source_children": 1,
        "replay_only_after_primary_success": True,
        "retry_rescue_or_cap_extension_permitted": False,
        "manual_native_review_required_when_visual_payload_exists": True,
        "manual_review_may_override_failure": False,
        "evaluator_roles_labels_gold_expected_counts_or_quality_access_permitted": False,
        "generalization_evidence": False,
        "accuracy_precision_recall_or_f1_authorized": False,
        "quality_rank_authorized": False,
        "bend_line_f1_authorized": False,
        "production_promotion_authorized": False,
    }
    return lock, row


def test_detached_approval_rejects_extra_keys_and_contradictory_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, row = _valid_detached_approval_fixture()
    approval = tmp_path / "approval.json"
    approval.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(contract, "_exact_cli_path", lambda *args, **kwargs: approval)
    monkeypatch.setattr(contract, "sha256_path", lambda _path: "c" * 64)
    monkeypatch.setattr(contract, "_load_json", lambda _path: row)
    contract._verify_detached_approval(
        approval,
        "c" * 64,
        lock_path=Path("/immutable-lock.json"),
        lock_sha256="b" * 64,
        lock=lock,
    )

    row["notes"] = "malicious extra authority surface"
    with pytest.raises(ValueError, match="field inventory"):
        contract._verify_detached_approval(
            approval,
            "c" * 64,
            lock_path=Path("/immutable-lock.json"),
            lock_sha256="b" * 64,
            lock=lock,
        )
    del row["notes"]

    for field in (
        "evaluator_input_authorized",
        "evaluator_roles_labels_gold_expected_counts_or_quality_access_permitted",
        "generalization_evidence",
        "accuracy_precision_recall_or_f1_authorized",
        "quality_rank_authorized",
        "bend_line_f1_authorized",
        "production_promotion_authorized",
    ):
        row[field] = True
        with pytest.raises(ValueError):
            contract._verify_detached_approval(
                approval,
                "c" * 64,
                lock_path=Path("/immutable-lock.json"),
                lock_sha256="b" * 64,
                lock=lock,
            )
        row[field] = False

    row["authorized_orchestrator_invocations"] = 2
    with pytest.raises(ValueError, match="budget"):
        contract._verify_detached_approval(
            approval,
            "c" * 64,
            lock_path=Path("/immutable-lock.json"),
            lock_sha256="b" * 64,
            lock=lock,
        )


def test_execution_authority_rejects_empty_or_mixed_path_maps_before_file_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        contract,
        "_exact_cli_path",
        lambda *args, **kwargs: pytest.fail("path access followed malformed maps"),
    )
    common = {
        "preregistration_path": Path("/prereg.json"),
        "lock_path": Path("/lock.json"),
        "lock_sha256": "a" * 64,
        "detached_approval_path": Path("/approval.json"),
        "detached_approval_sha256": "b" * 64,
    }
    for consumed, outputs in (
        ({}, {}),
        ({key: Path("/x") for key in contract.SOURCE_CONSUMED_PATH_KEYS}, {}),
        (
            {key: Path("/x") for key in contract.ORCHESTRATOR_CONSUMED_PATH_KEYS},
            {key: Path("/y") for key in contract.SOURCE_OUTPUT_PATH_KEYS},
        ),
    ):
        with pytest.raises(ValueError, match="inventories"):
            contract.verify_execution_authority(
                **common, consumed_paths=consumed, output_paths=outputs
            )


def test_audit_and_prelock_require_exact_test_evidence_and_counter_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_ids = ["test_one"]
    monkeypatch.setattr(contract, "EXPECTED_SOURCE_FREE_TEST_COUNT", 1)
    monkeypatch.setattr(
        contract,
        "EXPECTED_SOURCE_FREE_TEST_IDS_SHA256",
        hashlib.sha256(contract.canonical_bytes(test_ids)).hexdigest(),
    )
    result = {
        "collected_count": 1,
        "passed_count": 1,
        "failed_count": 0,
        "error_count": 0,
        "skipped_count": 0,
        "returncode": 0,
    }
    evidence = {
        "collect_command": list(contract.SOURCE_FREE_COLLECT_COMMAND),
        "test_command": list(contract.SOURCE_FREE_TEST_COMMAND),
        "collected_test_ids": test_ids,
    }
    counters = {key: 0 for key in contract.FORBIDDEN_ACCESS_COUNT_KEYS}
    audit = {
        "schema": contract.AUDIT_PACKET_SCHEMA,
        "experiment_id": contract.EXPERIMENT_ID,
        "status": "PASS_SOURCE_FREE_AUDIT_EXECUTION_NOT_AUTHORIZED",
        "physical_source_execution_authorized": False,
        "evaluator_input_authorized": False,
        "physical_scan_or_evaluator_input_used": False,
        "generalization_evidence": False,
        "source_free_regression": result,
        "forbidden_access_counts": counters,
        **evidence,
        "audit_findings": ["exact source-free authority audit passed"],
        "audited_file_sha256": {},
    }
    contract._validate_audit_packet(audit)
    bad_audit = json.loads(json.dumps(audit))
    bad_audit["forbidden_access_counts"] = {}
    with pytest.raises(ValueError, match="inventory"):
        contract._validate_audit_packet(bad_audit)

    digest = "a" * 64
    bound = {
        key: {
            "path": contract.PREREGISTRATION_PATH,
            "size_bytes": PREREG.stat().st_size,
            "sha256": digest,
        }
        for key in contract.REQUIRED_BOUND_FILES
    }
    test_attestation = {
        "result": result,
        **evidence,
    }
    monkeypatch.setattr(contract, "_load_json", lambda _path: test_attestation)
    prelock = {
        "schema": contract.PRELOCK_APPROVAL_SCHEMA,
        "experiment_id": contract.EXPERIMENT_ID,
        "status": "APPROVED_TO_MATERIALIZE_IMMUTABLE_PHYSICAL_LOCK",
        "immutable_lock_materialization_authorized": True,
        "physical_source_execution_authorized": False,
        "physical_orchestrator_invocation_authorized": False,
        "evaluator_input_authorized": False,
        "physical_scan_or_evaluator_input_used": False,
        "detached_postlock_physical_execution_approval_still_required": True,
        "generalization_evidence": False,
        "source_free_test_result": result,
        "forbidden_access_counts": counters,
        **evidence,
        "approved_bound_file_sha256": {
            key: digest
            for key in contract.REQUIRED_BOUND_FILES
            if key != "prelock_readiness_approval"
        },
    }
    contract._validate_prelock_approval(prelock, bound=bound)
    bad_prelock = json.loads(json.dumps(prelock))
    bad_prelock["collected_test_ids"] = []
    with pytest.raises(ValueError, match="test IDs"):
        contract._validate_prelock_approval(bad_prelock, bound=bound)
