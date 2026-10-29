from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_f1_v2_cross_parent_rung_mapping_preflight import (
    MAPPING_PREFLIGHT_CRITERIA,
    OUTCOME_ABSENT,
    PROFILE,
    RELATION_STATUS_ABSENT,
)
from scripts.validate_f1_v2_cross_parent_rung_mapping_preflight import (
    FROZEN_TARGET_FIELDS,
    PREFLIGHT_KIND,
    PREFLIGHT_TOP_FIELDS,
    _record_hash,
    _sha_json,
    _assert_criteria_match_derived,
    _assert_exact_all_pair_memberships,
    _assert_exact_target_preservation,
    _assert_exact_threshold_hash,
    _exact_frozen_target_projection,
    _validate_outcome,
    _validate_relation,
    validate_and_compact,
)


def _absent_relation() -> dict:
    modes = ["M_1", "M_2"]
    target = {
        "case_token": "C001",
        "frozen_target_component_id": "LTC_1",
        "frozen_target_pair_id": "LPP_1",
        "frozen_target_parent_id": "P_1",
        "frozen_target_mode_ids": modes,
        "frozen_target_mode_ids_sha256": _sha_json(modes),
    }
    target["frozen_target_sha256"] = _sha_json(target)
    mode_fragments = [["M_1", "VEF_L"], ["M_2", "VEF_R"]]
    fragments = ["VEF_L", "VEF_R"]
    row = {
        **target,
        "baseline_mode_fragment_rows": mode_fragments,
        "baseline_mode_fragment_rows_sha256": _sha_json(mode_fragments),
        "baseline_fragment_ids": fragments,
        "baseline_fragment_ids_sha256": _sha_json(fragments),
        "baseline_pair_mapping_count": 1,
        "baseline_local_plane_pair_id": "VEP_1",
        "baseline_pair_sha256": "1" * 64,
        "baseline_transverse_parent_count": 10,
        "baseline_transverse_parent_ids_sha256": "2" * 64,
        "target_parent_line_distance_mm": 7.421545400596292,
        "line_tube_tolerance_mm": 3.0,
        "target_parent_in_frozen_transverse_set": False,
        "baseline_component_mapping_count": 0,
        "complete_edge_baseline_component_id": None,
        "target_baseline_relation_status": RELATION_STATUS_ABSENT,
    }
    row["target_baseline_relation_sha256"] = _record_hash(
        row, "target_baseline_relation_sha256"
    )
    return row


def _absent_outcome(relation: dict) -> dict:
    row = {
        "case_token": relation["case_token"],
        "frozen_target_component_id": relation["frozen_target_component_id"],
        "frozen_target_mode_ids_sha256": relation["frozen_target_mode_ids_sha256"],
        "baseline_local_plane_pair_id": relation["baseline_local_plane_pair_id"],
        "target_baseline_relation_status": RELATION_STATUS_ABSENT,
        "complete_edge_baseline_component_id": None,
        "replacement_component_id": None,
        "causal_added_cross_parent_rung_count": 0,
        "causal_added_cross_parent_rung_ids_sha256": _sha_json([]),
        "support_parent_count_before": None,
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
        "target_outcome_code": OUTCOME_ABSENT,
        "target_reached_by_added_rung": False,
        "target_side_gate_advanced": False,
        "target_novel_rung_attributable": False,
    }
    row["target_outcome_sha256"] = _record_hash(row, "target_outcome_sha256")
    return row


def test_amended_criteria_are_exact_twenty_and_mapping_only() -> None:
    assert len(MAPPING_PREFLIGHT_CRITERIA) == 20
    assert MAPPING_PREFLIGHT_CRITERIA[-1] == "gold_open_authorized_false"
    assert "ordinary_batch_authorized" not in MAPPING_PREFLIGHT_CRITERIA


def test_out_of_tube_relation_and_outcome_are_structural_negative() -> None:
    relation = _absent_relation()
    outcome = _absent_outcome(relation)
    _validate_relation(relation)
    _validate_outcome(outcome, relation)
    assert relation["complete_edge_baseline_component_id"] is None
    assert relation["baseline_component_mapping_count"] == 0
    assert outcome["target_reached_by_added_rung"] is False
    assert outcome["target_side_gate_advanced"] is False
    assert outcome["target_novel_rung_attributable"] is False


def test_out_of_tube_outcome_rejects_imputed_zero_geometry() -> None:
    relation = _absent_relation()
    outcome = _absent_outcome(relation)
    outcome["support_parent_count_before"] = 0
    outcome["target_outcome_sha256"] = _record_hash(
        outcome, "target_outcome_sha256"
    )
    with pytest.raises(ValueError, match="must be null, not zero"):
        _validate_outcome(outcome, relation)


def test_unrecognised_relation_state_fails_closed() -> None:
    relation = _absent_relation()
    relation["target_baseline_relation_status"] = "THIRD_STATE"
    relation["target_baseline_relation_sha256"] = _record_hash(
        relation, "target_baseline_relation_sha256"
    )
    with pytest.raises(ValueError, match="unrecognised"):
        _validate_relation(relation)


def test_v1_preflight_identity_is_rejected_before_source_reads(tmp_path: Path) -> None:
    artifact = {field: None for field in PREFLIGHT_TOP_FIELDS}
    artifact.update({
        "schema_version": 1,
        "kind": PREFLIGHT_KIND,
        "profile": "V2_CROSS_PARENT_OPPOSITE_FRAGMENT_RUNG_PROBE_V1",
        "target_baseline_relation_rule_id": "OLD",
    })
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    missing = tmp_path / "must_not_be_read.json"
    with pytest.raises(ValueError, match="V1 is rejected"):
        validate_and_compact(
            preflight_path=path,
            expected_preflight_sha256=digest,
            inference_manifest_path=missing,
            candidate_pool_manifest_path=missing,
            a0_prediction_root=tmp_path,
            frozen_control_manifest_path=missing,
            complete_edge_baseline_manifest_path=missing,
        )


def test_validator_cli_exposes_no_unauthorized_lane_arguments() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/validate_f1_v2_cross_parent_rung_mapping_preflight.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
        check=True,
    )
    help_text = completed.stdout
    for forbidden in (
        "--gold", "--bundle", "--batch", "--replay", "--synthetic"
    ):
        assert forbidden not in help_text


def _pair_row(pair_id: str, parents: list[str]) -> dict:
    parents = sorted(parents)
    return {
        "baseline_local_plane_pair_id": pair_id,
        "transverse_parent_ids": parents,
        "transverse_parent_ids_sha256": _sha_json(parents),
    }


def test_tamper_non_target_pair_transverse_parent_is_rejected() -> None:
    expected = [
        _pair_row("VEP_TARGET", ["P_1", "P_2"]),
        _pair_row("VEP_NOT_REACHED_BY_TARGET", ["P_3", "P_4"]),
    ]
    tampered = [dict(row) for row in expected]
    tampered[1] = _pair_row("VEP_NOT_REACHED_BY_TARGET", ["P_3", "P_5"])
    with pytest.raises(ValueError, match="complete all-pair membership table mismatch"):
        _assert_exact_all_pair_memberships(
            tampered, expected, _sha_json(tampered)
        )


def test_tamper_target_id_exchange_with_unchanged_count_is_rejected() -> None:
    relation = _absent_relation()
    expected = _exact_frozen_target_projection([relation])
    tampered = dict(relation)
    tampered["frozen_target_component_id"] = "LTC_EXCHANGED"
    target = {
        field: tampered[field]
        for field in FROZEN_TARGET_FIELDS[:-1]
    }
    tampered["frozen_target_sha256"] = _sha_json(target)
    with pytest.raises(ValueError, match="ID sets are not exactly equal"):
        _assert_exact_target_preservation([tampered], expected)


def test_tamper_target_mode_hash_exchange_with_unchanged_count_is_rejected() -> None:
    relation = _absent_relation()
    expected = _exact_frozen_target_projection([relation])
    tampered = dict(relation)
    tampered["frozen_target_mode_ids"] = ["M_3", "M_4"]
    tampered["frozen_target_mode_ids_sha256"] = _sha_json(
        tampered["frozen_target_mode_ids"]
    )
    target = {
        field: tampered[field]
        for field in FROZEN_TARGET_FIELDS[:-1]
    }
    tampered["frozen_target_sha256"] = _sha_json(target)
    with pytest.raises(ValueError, match="mode-hash sets are not exactly equal"):
        _assert_exact_target_preservation([tampered], expected)


def test_tamper_valid_looking_wrong_threshold_hash_is_rejected() -> None:
    with pytest.raises(ValueError, match="threshold hash changed"):
        _assert_exact_threshold_hash("a" * 64, "b" * 64)


def test_true_pair_criterion_over_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="not source-derived"):
        _assert_criteria_match_derived(
            {"all_pair_line_tube_memberships_are_unchanged": True},
            {"all_pair_line_tube_memberships_are_unchanged": False},
        )
