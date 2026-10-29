from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import inspect
import subprocess
import sys

import pytest

from backend import f1_v2_cross_parent_opposite_fragment_rung_probe_v2 as core
from scripts.validate_f1_v2_cross_parent_ordinary_batch import (
    DELETION_TABLE_KEYS,
    _assert_criteria_derived,
    _assert_deletion_input_binding,
    _assert_executed_deletion_restoration,
    _assert_external_sha,
    _assert_exact_table,
    _assert_structural_negative_outcome,
    _assert_pre_rung_gate_derived,
    _sha_json,
    _walk_forbidden,
    _ordered_case_summaries,
    canonical_evidence_bytes,
    validate_and_compact,
)
from scripts import run_f1_v2_cross_parent_opposite_fragment_rung_ordinary_batch_v2 as runner


def _deletion_tables() -> dict[str, list[dict]]:
    return {
        name: [{"id": f"{name}:1", "value": 1}]
        for name in DELETION_TABLE_KEYS
    }


def test_external_authoritative_sha_mismatch_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "ordinary.json"
    artifact.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="external SHA mismatch"):
        _assert_external_sha(artifact, "0" * 64)


def test_forged_deletion_true_over_mismatching_raw_table_rejected() -> None:
    frozen = _deletion_tables()
    rebuilt = deepcopy(frozen)
    rebuilt["geometry_gate_table"][0]["value"] = 2
    with pytest.raises(ValueError, match="raw canonical tables"):
        _assert_executed_deletion_restoration(
            rebuilt_tables=rebuilt,
            frozen_source_tables=frozen,
            asserted_valid=True,
        )


def test_v1_hash_copy_deletion_pattern_rejected_without_raw_tables() -> None:
    copied_hash_only = {
        "restored_complete_edge_table_sha256": "1" * 64,
        "expected_complete_edge_table_sha256": "1" * 64,
        "fragment_table_restored": True,
        "orphan_table_restored": True,
        "pair_component_records_restored": True,
        "global_deletion_counterfactual_valid": True,
    }
    with pytest.raises(ValueError, match="rebuilt-table schema mismatch"):
        _assert_executed_deletion_restoration(
            rebuilt_tables=copied_hash_only,
            frozen_source_tables=_deletion_tables(),
            asserted_valid=True,
        )


def test_deletion_count_only_does_not_bind_exact_added_rung_table() -> None:
    added = [{"mode_edge_id": "E_1"}, {"mode_edge_id": "E_2"}]
    with pytest.raises(ValueError, match="does not bind ordinary input"):
        _assert_deletion_input_binding(
            added_rung_table=added,
            deleted_rung_count=2,
            deleted_rung_table_sha256="f" * 64,
        )


def test_structural_negative_rejects_zero_imputation() -> None:
    row = {
        "complete_edge_baseline_component_id": None,
        "replacement_component_id": None,
        "causal_added_cross_parent_rung_count": 0,
        "causal_added_cross_parent_rung_ids_sha256": _sha_json([]),
        "support_parent_count_before": 0,
        "support_parent_count_after": None,
        "occupied_bin_count_before": None,
        "occupied_bin_count_after": None,
        "left_side_bin_count_before": None,
        "left_side_bin_count_after": None,
        "right_side_bin_count_before": None,
        "right_side_bin_count_after": None,
        "transition_rung_bin_count_before": None,
        "transition_rung_bin_count_after": None,
        "geometry_gate_depth_before": None,
        "geometry_gate_depth_after": None,
        "first_rejection_code_before": None,
        "first_rejection_code_after": None,
        "geometry_certificate_valid_after": False,
        "target_reached_by_added_rung": False,
        "target_side_gate_advanced": False,
        "target_novel_rung_attributable": False,
        "target_outcome_code": core.OUTCOME_STRUCTURALLY_UNREACHABLE,
    }
    with pytest.raises(ValueError, match="null, not zero"):
        _assert_structural_negative_outcome(row)


def test_forged_true_criterion_over_false_derived_predicate_rejected() -> None:
    asserted = {key: True for key in core.ORDINARY_V2_CRITERION_KEYS}
    derived = dict(asserted)
    derived["global_deletion_counterfactual_executed_and_valid"] = False
    with pytest.raises(ValueError, match="not derived"):
        _assert_criteria_derived(asserted, derived)


def test_complete_table_tamper_with_rehashed_row_but_wrong_identity_rejected() -> None:
    rows = [{"case_token": "C001", "id": "A"}, {"case_token": "C001", "id": "B"}]
    tampered = deepcopy(rows)
    tampered[1]["id"] = "A"
    with pytest.raises(ValueError, match="count/order/identity"):
        _assert_exact_table(
            tampered,
            expected_count=2,
            id_fields=("case_token", "id"),
            declared_sha256=_sha_json(tampered),
            label="attribution table",
        )


def test_forbidden_gold_or_runtime_payload_key_rejected() -> None:
    with pytest.raises(ValueError, match="forbidden ordinary-V2 key"):
        _walk_forbidden({"nested": {"gold": []}})
    with pytest.raises(ValueError, match="forbidden ordinary-V2 key"):
        _walk_forbidden({"runtime_integration": True})


def test_validator_cli_has_no_unauthorized_lane_arguments() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/validate_f1_v2_cross_parent_ordinary_batch.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"], text=True,
        capture_output=True, check=True,
    )
    for forbidden in ("--gold", "--replay", "--synthetic", "--runtime", "--batch"):
        assert forbidden not in completed.stdout


def test_copied_global_pre_rung_criteria_rejected_when_case_evidence_fails() -> None:
    copied_global = {key: True for key in runner.MAPPING_PREFLIGHT_CRITERIA}
    derived_case = dict(copied_global)
    derived_case["all_pair_line_tube_memberships_are_unchanged"] = False
    with pytest.raises(ValueError, match="copied, forged, or underived"):
        _assert_pre_rung_gate_derived(
            copied_global, derived_case, _sha_json(copied_global)
        )


def test_worker_cannot_construct_rungs_before_case_gate_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = [tmp_path / f"source-{index}.json" for index in range(5)]
    for index, path in enumerate(paths):
        path.write_bytes(runner.canonical_json_bytes({"source": index}))
    hashes = [runner.sha256_file(path) for path in paths]
    expected_mapping = {
        "frozen_target_projection": [],
        "frozen_target_baseline_relation_table": [],
        "frozen_target_outcome_table": [],
        "all_pair_membership_table": [],
        "thresholds_sha256": _sha_json({}),
        "mapping_case_summary": {"frozen_target_count": 0},
    }
    monkeypatch.setattr(runner.core, "project_mapping_preflight_input", lambda _raw: {})
    monkeypatch.setattr(
        runner.complete_core, "regenerate_frozen_control_probe",
        lambda *_args, **_kwargs: {"source": 3},
    )
    monkeypatch.setattr(
        runner.complete_core, "build_v2_complete_same_surface_edge_probe",
        lambda *_args, **_kwargs: {"source": 4},
    )
    monkeypatch.setattr(
        runner.core, "_build_case_from_payloads",
        lambda *_args, **_kwargs: {"mapping": True},
    )
    monkeypatch.setattr(
        runner.core, "validate_v2_cross_parent_mapping_case_payload",
        lambda *_args, **_kwargs: None,
    )
    failed_gate = {key: True for key in runner.MAPPING_PREFLIGHT_CRITERIA}
    failed_gate["all_pair_line_tube_memberships_are_unchanged"] = False
    monkeypatch.setattr(
        runner, "_derive_pre_rung_gate", lambda **_kwargs: failed_gate
    )
    ordinary_called = False

    def forbidden_ordinary_build(*_args, **_kwargs):
        nonlocal ordinary_called
        ordinary_called = True
        raise AssertionError("ordinary rung construction ran before gate")

    monkeypatch.setattr(
        runner.core, "build_v2_cross_parent_ordinary_case",
        forbidden_ordinary_build,
    )
    task = (
        "C001", *(str(path) for path in paths), *hashes, expected_mapping,
    )
    with pytest.raises(ValueError, match="per-case pre-rung gate failed"):
        runner._run_case(task)
    assert ordinary_called is False


def test_independent_validator_worker_bound_fails_before_io(tmp_path: Path) -> None:
    kwargs = {
        "authoritative_artifact_path": tmp_path / "missing.json",
        "expected_authoritative_sha256": "0" * 64,
        "mapping_preflight_path": tmp_path / "mapping.json",
        "mapping_compact_evidence_path": tmp_path / "compact.json",
        "inference_manifest_path": tmp_path / "inference.json",
        "candidate_pool_manifest_path": tmp_path / "pool.json",
        "a0_prediction_root": tmp_path / "a0",
        "frozen_control_manifest_path": tmp_path / "control.json",
        "complete_edge_baseline_manifest_path": tmp_path / "baseline.json",
    }
    with pytest.raises(ValueError, match=r"workers must be in \[1,8\]"):
        validate_and_compact(**kwargs, workers=0)
    with pytest.raises(ValueError, match=r"workers must be in \[1,8\]"):
        validate_and_compact(**kwargs, workers=9)


def test_independent_validator_uses_fresh_process_per_case_and_small_tasks() -> None:
    source = inspect.getsource(validate_and_compact)
    assert "ProcessPoolExecutor" in source
    assert "max_workers=workers" in source
    assert "max_tasks_per_child=1" in source
    task_body = source.split("tasks.append((", 1)[1].split("))", 1)[0]
    assert "embedded_case_payloads" not in task_body
    assert "deepcopy(manifest_row)" in task_body
    assert "deepcopy(manifest)," not in task_body


def test_worker_completion_order_cannot_change_compact_bytes() -> None:
    rows = [
        {"case_token": token, "result_sha256": f"{index:064x}"}
        for index, token in enumerate(runner.EXPECTED_CASE_TOKENS, start=1)
    ]
    forward = _ordered_case_summaries(rows)
    reverse = _ordered_case_summaries(list(reversed(rows)))
    assert forward == reverse
    assert canonical_evidence_bytes({"case_summaries": forward}) == (
        canonical_evidence_bytes({"case_summaries": reverse})
    )
