from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import exp_cad_0034_execution_contract as contract


PREREG = Path(
    "docs/research/"
    "2026-08-30_exp_cad_0034_cfpdg_v5_physical_successor_preregistration.json"
)


def test_frozen_preregistration_identity_and_digest() -> None:
    block = PREREG.read_bytes()
    assert hashlib.sha256(block).hexdigest() == contract.PREREGISTRATION_SHA256
    value = json.loads(block)
    assert value["schema"] == "EXP_CAD_0034_CFPDG_V5_PHYSICAL_SUCCESSOR_PREREGISTRATION_V1"
    assert value["experiment_id"] == "EXP_CAD_0034"
    assert value["preregistration_boundary"]["physical_scan_hash_call_count_during_exp34_design"] == 0
    assert value["preregistration_boundary"]["physical_scan_point_cloud_load_call_count_during_exp34_design"] == 0
    assert value["preregistration_boundary"]["physical_scan_byte_open_operation_count_during_exp34_design"] == 0


def test_scan_rows_are_expected_constants_not_file_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))

    def forbidden_hash(_path: Path) -> str:
        raise AssertionError("physical scan hash must not run while constructing lock rows")

    monkeypatch.setattr(contract, "sha256_path", forbidden_hash)
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
    row = {
        "schema": contract.TEST_ATTESTATION_SCHEMA,
        "experiment_id": contract.EXPERIMENT_ID,
        "status": "PASS_SOURCE_FREE_TESTS",
        "result": {
            "collected_count": 1,
            "passed_count": 1,
            "failed_count": 0,
            "error_count": 0,
            "skipped_count": 0,
            "returncode": 0,
        },
        "forbidden_access_counts": {"physical_scan_hash_call_count": 0},
        "collected_test_ids": ["test_one"],
        "tested_file_sha256": {"test": "0" * 64},
    }
    contract._validate_test_attestation(row)
    skipped = json.loads(json.dumps(row))
    skipped["result"]["skipped_count"] = 1
    with pytest.raises(ValueError):
        contract._validate_test_attestation(skipped)
    opened = json.loads(json.dumps(row))
    opened["forbidden_access_counts"]["physical_scan_hash_call_count"] = 1
    with pytest.raises(ValueError):
        contract._validate_test_attestation(opened)


def test_environment_must_equal_frozen_exp33_environment() -> None:
    parent = json.loads(
        Path("docs/research/2026-08-30_exp_cad_0033_source_free_environment_attestation.json").read_text(
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
        "environment": parent["environment"],
    }
    contract._validate_environment_attestation(row, parent)
    changed = json.loads(json.dumps(row))
    changed["environment"]["open3d_version"] = "different"
    with pytest.raises(ValueError):
        contract._validate_environment_attestation(changed, parent)


def test_live_runtime_fingerprint_matches_frozen_exp33_environment() -> None:
    parent = json.loads(
        Path("docs/research/2026-08-30_exp_cad_0033_source_free_environment_attestation.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract.runtime_environment_fingerprint() == parent["environment"]


def test_lock_builder_requires_exact_key_set_without_touching_inputs() -> None:
    with pytest.raises(ValueError, match="lock keys differ"):
        contract.build_lock_payload(bindings={}, lock_path=contract.ROOT / contract.LOCK_PATH)
